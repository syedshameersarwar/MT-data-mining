"""
t-SNE Analysis Module for Cardiac Signal Feature Dimensionality Reduction

This module provides comprehensive t-SNE (t-Distributed Stochastic Neighbor Embedding)
analysis for cardiac signal features across different drug treatments. The analysis
focuses on visualizing high-dimensional feature relationships in 2D space to identify
drug-specific effects and tissue-level patterns.

Scientific Background:
    t-SNE is a nonlinear dimensionality reduction technique that:
    1. Preserves local structure and clusters in high-dimensional data
    2. Reveals patterns that may be hidden in the original feature space
    3. Enables visualization of complex relationships between cardiac parameters
    4. Helps identify drug-specific effects on cardiac signal coordination

    In cardiac electrophysiology, t-SNE analysis reveals:
    - Drug-specific clustering patterns in feature space
    - Tissue-level variability in drug responses
    - Concentration-dependent effects on cardiac function
    - Relationships between electrical, calcium, and mechanical parameters

Key Analysis Components:
    1. Data Preparation: Feature selection and normalization
    2. t-SNE Dimensionality Reduction: 2D embedding with configurable parameters
    3. Averaging: Tissue-concentration level aggregation for clarity
    4. Statistical Analysis: Cluster analysis and pattern identification

Pipeline Architecture:
    Data Loading → Feature Selection → t-SNE Embedding → Averaging → Analysis

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
    - t-SNE embeddings (2D coordinates)
    - Averaged tissue-concentration data points
    - Feature importance analysis
    - Statistical summaries of clustering patterns

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
Dependencies: pandas, numpy, scikit-learn, matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSneAnalyzer:
    """
    Class for performing t-SNE analysis on cardiac signal features.

    This class handles the complete t-SNE analysis pipeline including data preparation,
    dimensionality reduction, averaging, and statistical analysis. It provides methods
    for analyzing drug-specific effects on cardiac signal feature relationships.

    Attributes:
        random_state (int): Random seed for reproducible results
        n_components (int): Number of t-SNE components (default: 2)
        perplexity (float): t-SNE perplexity parameter
        scaler (StandardScaler): Feature scaler for normalization
        tsne_model (TSNE): Fitted t-SNE model
        metadata_cols (List[str]): Columns containing metadata (not features)
    """

    def __init__(
        self,
        random_state: int = 42,
        n_components: int = 2,
        perplexity: float = 30.0,
        standard_scaling: bool = True,
    ):
        """
        Initialize TSneAnalyzer with specified parameters.

        Args:
            random_state (int): Random seed for reproducible results
            n_components (int): Number of t-SNE components (default: 2)
            perplexity (float): t-SNE perplexity parameter (default: 30.0)
            standard_scaling (bool): Whether to apply standard scaling to features (default: True)
        """
        self.random_state = random_state
        self.n_components = n_components
        self.perplexity = perplexity
        self.standard_scaling = standard_scaling

        # Initialize components
        self.scaler = StandardScaler()
        self.tsne_model = None

        # Define metadata columns
        self.metadata_cols = [
            "drug",
            "bct_id",
            "concentration[um]",
            "frequency[Hz]",
            "data_type",
        ]

        logger.info(
            f"Initialized TSneAnalyzer with random_state={random_state}, "
            f"perplexity={perplexity}, standard_scaling={standard_scaling}"
        )

    def prepare_data(
        self,
        baseline_df: pd.DataFrame,
        drug_dfs: List[pd.DataFrame],
        drug_names: List[str],
        selected_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Prepare data for t-SNE analysis by combining and organizing datasets.

        This method combines baseline and drug treatment data, adds data type
        labels, and selects relevant features for analysis.

        Args:
            baseline_df (pd.DataFrame): Baseline measurements data
            drug_dfs (List[pd.DataFrame]): List of drug treatment dataframes
            drug_names (List[str]): List of drug names corresponding to drug_dfs
            selected_features (Optional[List[str]]): Subset of features to analyze

        Returns:
            pd.DataFrame: Combined and prepared data for t-SNE analysis

        Example:
            >>> analyzer = TSneAnalyzer()
            >>> combined_data = analyzer.prepare_data(
            ...     baseline_df=baseline_cases,
            ...     drug_dfs=[e4031_cases, nifedipine_cases, ca_titration_cases],
            ...     drug_names=["e4031", "nifedipine", "ca_titration"],
            ...     selected_features=["duration", "force_peak_amplitude", "calc_peak_amplitude"]
            ... )
        """
        logger.info("Preparing data for t-SNE analysis...")

        # Data preparation
        all_dfs = []

        # Add baseline data
        baseline_copy = baseline_df.copy()
        baseline_copy["data_type"] = "Baseline"
        all_dfs.append(baseline_copy)

        # Add drug data
        for df, drug_name in zip(drug_dfs, drug_names):
            drug_copy = df.copy()
            drug_copy["data_type"] = drug_name
            all_dfs.append(drug_copy)

        # Combine all data
        combined_data = pd.concat(all_dfs, ignore_index=True)

        # Feature selection
        feature_cols = [
            col for col in combined_data.columns if col not in self.metadata_cols
        ]

        if selected_features:
            feature_cols = [col for col in feature_cols if col in selected_features]
            logger.info(f"Selected {len(feature_cols)} features for analysis")

        # Ensure all required columns are present
        required_cols = self.metadata_cols + feature_cols
        missing_cols = [
            col for col in required_cols if col not in combined_data.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(
            f"Prepared data with {len(combined_data)} records and "
            f"{len(feature_cols)} features"
        )

        return combined_data, feature_cols

    def perform_tsne(
        self, data: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Perform t-SNE dimensionality reduction on the prepared data.

        This method applies t-SNE to reduce high-dimensional feature data to
        2D coordinates while preserving local structure and relationships.

        Args:
            data (pd.DataFrame): Combined data with features and metadata
            feature_cols (List[str]): List of feature column names

        Returns:
            Tuple[np.ndarray, pd.DataFrame]:
                - t-SNE results (2D coordinates)
                - Plot DataFrame with coordinates and metadata

        Example:
            >>> tsne_results, plot_df = analyzer.perform_tsne(combined_data, feature_cols)
        """
        logger.info("Performing t-SNE dimensionality reduction...")

        # Extract feature data
        feature_data = data[feature_cols].values

        # Scale features if requested
        if self.standard_scaling:
            logger.info("Applying standard scaling to features...")
            scaled_features = self.scaler.fit_transform(feature_data)
        else:
            logger.info("Skipping standard scaling, using raw features...")
            scaled_features = feature_data

        # Perform t-SNE
        logger.info(f"Running t-SNE with perplexity={self.perplexity}")

        self.tsne_model = TSNE(
            n_components=self.n_components,
            random_state=self.random_state,
            perplexity=self.perplexity,
        )

        tsne_results = self.tsne_model.fit_transform(scaled_features)

        # Create plot DataFrame with coordinates and metadata
        plot_df = pd.DataFrame(
            {
                "TSNE1": tsne_results[:, 0],
                "TSNE2": tsne_results[:, 1],
                "concentration": data["concentration[um]"],
                "data_type": data["data_type"],
                "bct_id": data["bct_id"],
                "drug": data["drug"],
                "frequency[Hz]": data["frequency[Hz]"],
            }
        )

        logger.info(
            f"t-SNE completed. Final KL divergence: {self.tsne_model.kl_divergence_:.4f}"
        )

        return tsne_results, plot_df

    def average_by_tissue_concentration(self, plot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Average t-SNE coordinates by tissue and concentration combination.

        This method aggregates data points to reduce noise and improve
        visualization clarity by averaging coordinates for each unique
        tissue-concentration combination.

        Args:
            plot_df (pd.DataFrame): DataFrame with t-SNE coordinates and metadata

        Returns:
            pd.DataFrame: Averaged data with one point per tissue-concentration

        Example:
            >>> averaged_df = analyzer.average_by_tissue_concentration(plot_df)
        """
        logger.info("Averaging t-SNE coordinates by tissue and concentration...")

        # Calculate averages for each tissue-concentration combination
        averaged_df = (
            plot_df.groupby(["data_type", "bct_id", "concentration"])
            .agg(
                {
                    "TSNE1": "mean",
                    "TSNE2": "mean",
                    "drug": "first",
                    "frequency[Hz]": "first",
                }
            )
            .reset_index()
        )

        logger.info(f"Averaged data from {len(plot_df)} to {len(averaged_df)} points")

        return averaged_df

    def analyze_clustering_patterns(self, averaged_df: pd.DataFrame) -> Dict:
        """
        Analyze clustering patterns in the t-SNE results.

        This method provides statistical analysis of the clustering patterns,
        including drug-specific clusters, concentration effects, and tissue
        variability.

        Args:
            averaged_df (pd.DataFrame): Averaged t-SNE data

        Returns:
            Dict: Statistical analysis results

        Example:
            >>> analysis_results = analyzer.analyze_clustering_patterns(averaged_df)
        """
        logger.info("Analyzing clustering patterns...")

        analysis_results = {}

        # Drug-specific analysis
        for data_type in averaged_df["data_type"].unique():
            subset = averaged_df[averaged_df["data_type"] == data_type]

            analysis_results[data_type] = {
                "n_points": int(len(subset)),
                "n_tissues": int(subset["bct_id"].nunique()),
                "n_concentrations": int(subset["concentration"].nunique()),
                "tsne1_range": (
                    float(subset["TSNE1"].min()),
                    float(subset["TSNE1"].max()),
                ),
                "tsne2_range": (
                    float(subset["TSNE2"].min()),
                    float(subset["TSNE2"].max()),
                ),
                "tsne1_std": float(subset["TSNE1"].std()),
                "tsne2_std": float(subset["TSNE2"].std()),
            }

        # Overall statistics
        analysis_results["overall"] = {
            "total_points": int(len(averaged_df)),
            "total_tissues": int(averaged_df["bct_id"].nunique()),
            "total_drugs": int(averaged_df["data_type"].nunique()),
            "tsne1_range": (
                float(averaged_df["TSNE1"].min()),
                float(averaged_df["TSNE1"].max()),
            ),
            "tsne2_range": (
                float(averaged_df["TSNE2"].min()),
                float(averaged_df["TSNE2"].max()),
            ),
        }

        logger.info("Clustering pattern analysis completed")

        return analysis_results

    def run_complete_analysis(
        self,
        baseline_df: pd.DataFrame,
        drug_dfs: List[pd.DataFrame],
        drug_names: List[str],
        selected_features: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run complete t-SNE analysis pipeline.

        This method orchestrates the entire t-SNE analysis workflow from data
        preparation through final analysis, providing comprehensive results
        for visualization and interpretation.

        Args:
            baseline_df (pd.DataFrame): Baseline measurements data
            drug_dfs (List[pd.DataFrame]): List of drug treatment dataframes
            drug_names (List[str]): List of drug names corresponding to drug_dfs
            selected_features (Optional[List[str]]): Subset of features to analyze

        Returns:
            Dict: Complete analysis results including:
                - combined_data: Original combined data
                - feature_cols: Selected feature columns
                - tsne_results: Raw t-SNE coordinates
                - plot_df: DataFrame with coordinates and metadata
                - averaged_df: Averaged tissue-concentration data
                - analysis_results: Statistical analysis results

        Example:
            >>> results = analyzer.run_complete_analysis(
            ...     baseline_df=baseline_cases,
            ...     drug_dfs=[e4031_cases, nifedipine_cases, ca_titration_cases],
            ...     drug_names=["e4031", "nifedipine", "ca_titration"]
            ... )
        """
        logger.info("Starting complete t-SNE analysis pipeline...")

        # Step 1: Prepare data
        combined_data, feature_cols = self.prepare_data(
            baseline_df, drug_dfs, drug_names, selected_features
        )

        # Step 2: Perform t-SNE
        tsne_results, plot_df = self.perform_tsne(combined_data, feature_cols)

        # Step 3: Average by tissue and concentration
        averaged_df = self.average_by_tissue_concentration(plot_df)

        # Step 4: Analyze clustering patterns
        analysis_results = self.analyze_clustering_patterns(averaged_df)

        # Compile results
        results = {
            "combined_data": combined_data,
            "feature_cols": feature_cols,
            "tsne_results": tsne_results,
            "plot_df": plot_df,
            "averaged_df": averaged_df,
            "analysis_results": analysis_results,
            "tsne_model": self.tsne_model,
            "scaler": self.scaler,
        }

        logger.info("Complete t-SNE analysis pipeline finished successfully")

        return results
