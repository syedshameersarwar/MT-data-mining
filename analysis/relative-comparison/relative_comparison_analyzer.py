"""
Relative Comparison Analysis Module for Cardiac Signal Feature Analysis

This module provides comprehensive analysis capabilities for comparing relative
feature changes between drug treatments and baseline conditions. The analyzer
handles three levels of comparison: tissue-specific, global average, and target
concentration analysis.

Scientific Background:
    Relative comparison analysis reveals:
    - Tissue-specific drug responses
    - Global drug effects across all tissues
    - Concentration-dependent feature changes
    - EC50/IC50 concentration effects

Key Analysis Components:
    1. Tissue-specific relative change calculation
    2. Global average feature change computation
    3. Target concentration comparison analysis
    4. Statistical summary generation
    5. Feature change quantification

Pipeline Architecture:
    Signal Data → Feature Change Calculation → Statistical Analysis → Results Export

Supported Analysis Types:
    - Tissue-level relative changes
    - Global average changes
    - Target concentration comparisons
    - EC50/IC50 concentration analysis

Drug Treatments Analyzed:
    - Baseline: Control conditions
    - E-4031: hERG potassium channel blocker
    - Nifedipine: L-type calcium channel blocker
    - Ca²⁺ Titration: Calcium concentration modulation

Output Products:
    - Relative change DataFrames
    - Statistical summaries
    - Feature change matrices
    - Analysis metadata

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
Dependencies: pandas, numpy, scipy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Import from utils to avoid duplication
from utils import SignalData, FeatureMapping

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelativeComparisonAnalyzer:
    """
    Class for analyzing relative feature changes between drug treatments and baseline.

    This class handles the calculation of relative percent differences between
    drug-treated and baseline conditions at three different levels:
    1. Tissue-specific: Individual tissue and concentration comparisons
    2. Global average: Averaged across all tissues and concentrations
    3. Target concentration: Specific concentration comparisons (e.g., EC50/IC50)

    Attributes:
        signal_data (SignalData): Processed signal data object
        feature_columns (List[str]): List of feature columns to analyze
        analysis_results (Dict): Dictionary storing all analysis results
    """

    def __init__(self, signal_data: SignalData):
        """
        Initialize RelativeComparisonAnalyzer with signal data.

        Args:
            signal_data (SignalData): Processed signal data object
        """
        self.signal_data = signal_data
        self.feature_columns = signal_data.features
        self.analysis_results = {}

        logger.info(
            f"Initialized RelativeComparisonAnalyzer with {len(self.feature_columns)} features"
        )

    def calculate_tissue_specific_changes(
        self, baseline_df: pd.DataFrame, drug_dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate tissue-specific relative feature changes compared to baseline.

        This method calculates relative percent differences for each tissue and
        concentration combination, providing detailed tissue-level drug responses.

        Args:
            baseline_df (pd.DataFrame): Baseline measurements
            drug_dfs (Dict[str, pd.DataFrame]): Dictionary of drug treatment dataframes

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - mean_features_df: DataFrame with mean feature values
                - relative_diff_df: DataFrame with relative percent differences

        Example:
            >>> analyzer = RelativeComparisonAnalyzer(signal_data)
            >>> drug_dfs = {
            ...     'e4031': e4031_cases,
            ...     'nifedipine': nifedipine_cases,
            ...     'ca_titration': ca_titration_cases
            ... }
            >>> mean_features, relative_diffs = analyzer.calculate_tissue_specific_changes(
            ...     baseline_df, drug_dfs
            ... )
        """
        logger.info("Calculating tissue-specific relative changes...")

        def calculate_means(df: pd.DataFrame) -> pd.DataFrame:
            """Calculate means for each tissue, drug, and concentration combination"""
            metadata_cols = ["drug", "bct_id", "concentration[um]"]
            return df.groupby(metadata_cols)[self.feature_columns].mean().reset_index()

        # Calculate means for baseline and all drugs
        baseline_means = calculate_means(baseline_df)
        drug_means = {}
        for drug_name, drug_df in drug_dfs.items():
            drug_means[drug_name] = calculate_means(drug_df)

        # Combine all means
        all_means = pd.concat([baseline_means] + list(drug_means.values()))

        # Calculate relative differences
        relative_differences = pd.DataFrame()

        for drug_name, drug_means_df in drug_means.items():
            drug_changes = []

            for _, drug_row in drug_means_df.iterrows():
                # Find matching baseline record
                baseline_match = baseline_means[
                    (baseline_means["bct_id"] == drug_row["bct_id"])
                    & (baseline_means["drug"] == drug_row["drug"])
                ]

                if not baseline_match.empty:
                    baseline_values = baseline_match.iloc[0]

                    # Calculate relative differences for all features
                    changes = {
                        "bct_id": drug_row["bct_id"],
                        "drug": drug_row["drug"],
                        "concentration[um]": drug_row["concentration[um]"],
                    }

                    for feature in self.feature_columns:
                        if feature == "frequency[Hz]":
                            continue

                        baseline_value = baseline_values[feature]
                        if baseline_value != 0:  # Avoid division by zero
                            pct_change = np.round(
                                (
                                    (drug_row[feature] - baseline_value)
                                    / abs(baseline_value)
                                )
                                * 100,
                                2,
                            )
                        else:
                            pct_change = np.nan
                        changes[feature] = pct_change

                    drug_changes.append(changes)

            if drug_changes:
                drug_changes_df = pd.DataFrame(drug_changes)
                relative_differences = pd.concat(
                    [relative_differences, drug_changes_df]
                )

        relative_differences = relative_differences.reset_index(drop=True)

        logger.info(
            f"Calculated tissue-specific changes for {len(relative_differences)} drug-tissue-concentration combinations"
        )

        return all_means, relative_differences

    def calculate_global_average_changes(
        self, baseline_df: pd.DataFrame, drug_dfs: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate global average feature changes across all tissues and concentrations.

        This method provides a high-level view of drug effects by averaging
        across all tissue and concentration combinations.

        Args:
            baseline_df (pd.DataFrame): Baseline measurements
            drug_dfs (Dict[str, pd.DataFrame]): Dictionary of drug treatment dataframes

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - mean_features_df: DataFrame with global mean feature values
                - relative_diff_df: DataFrame with global relative differences
        """
        logger.info("Calculating global average feature changes...")

        def calculate_global_means(df: pd.DataFrame) -> pd.Series:
            """Calculate means across all tissues and concentrations"""
            return df[self.feature_columns].mean()

        # Calculate global means for baseline and each drug
        baseline_means = calculate_global_means(baseline_df)
        drug_means = {}

        for drug_name, drug_df in drug_dfs.items():
            drug_means[drug_name] = calculate_global_means(drug_df)

        # Combine all means into one dataframe
        mean_features_df = pd.DataFrame({"baseline": baseline_means, **drug_means})

        # Calculate relative differences for each drug
        relative_diff_df = pd.DataFrame()

        for drug_name, drug_means_series in drug_means.items():
            changes = {"drug": drug_name}

            for feature in self.feature_columns:
                if feature == "frequency[Hz]":
                    continue

                baseline_value = baseline_means[feature]
                if baseline_value != 0:
                    pct_change = np.round(
                        (
                            (drug_means_series[feature] - baseline_value)
                            / abs(baseline_value)
                        )
                        * 100,
                        2,
                    )
                else:
                    pct_change = np.nan
                changes[feature] = pct_change

            relative_diff_df = pd.concat([relative_diff_df, pd.DataFrame([changes])])

        relative_diff_df = relative_diff_df.reset_index(drop=True)

        logger.info("Global average feature changes calculated successfully")

        return mean_features_df, relative_diff_df

    def calculate_target_concentration_changes(
        self,
        baseline_df: pd.DataFrame,
        drug_dfs: Dict[str, pd.DataFrame],
        target_concentrations: List[float],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate relative feature changes at specific target concentrations.

        This method is particularly useful for EC50/IC50 concentration analysis,
        comparing baseline to specific concentrations of each drug.

        Args:
            baseline_df (pd.DataFrame): Baseline measurements
            drug_dfs (Dict[str, pd.DataFrame]): Dictionary of drug treatment dataframes
            target_concentrations (List[float]): List of target concentrations [drug1_conc, drug2_conc, drug3_conc]

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - mean_features_df: DataFrame with mean feature values at target concentrations
                - relative_diff_df: DataFrame with relative differences compared to baseline

        Example:
            >>> target_conc = [0.03, 1.0, 1.0]  # EC50/IC50 concentrations
            >>> mean_features, relative_diffs = analyzer.calculate_target_concentration_changes(
            ...     baseline_df, drug_dfs, target_conc
            ... )
        """
        logger.info(
            f"Calculating target concentration changes for concentrations: {target_concentrations}"
        )

        def calculate_conc_means(df: pd.DataFrame, target: float) -> pd.Series:
            """Calculate means at specific concentration"""
            return df[df["concentration[um]"] == target][self.feature_columns].mean()

        def calculate_global_means(df: pd.DataFrame) -> pd.Series:
            """Calculate means across all tissues and concentrations"""
            return df[self.feature_columns].mean()

        # Get drug names and calculate baseline means
        baseline_means = calculate_global_means(baseline_df)
        drug_names = list(drug_dfs.keys())

        # Calculate means at target concentrations for each drug
        drug_means = {}
        for i, (drug_name, drug_df) in enumerate(drug_dfs.items()):
            if i < len(target_concentrations):
                target_conc = target_concentrations[i]
                drug_means[drug_name] = calculate_conc_means(drug_df, target_conc)

        # Combine all means into one dataframe with concentration labels
        mean_features_df = pd.DataFrame(
            {
                "baseline": baseline_means,
                **{
                    f"{drug_name} ({target_concentrations[i]} µM)": drug_means[
                        drug_name
                    ]
                    for i, drug_name in enumerate(drug_names)
                    if i < len(target_concentrations)
                },
            }
        )

        # Calculate relative differences for each drug
        relative_diff_df = pd.DataFrame()

        for i, (drug_name, drug_means_series) in enumerate(drug_means.items()):
            if i < len(target_concentrations):
                conc = target_concentrations[i]
                changes = {"drug": f"{drug_name} ({conc} µM)"}

                for feature in self.feature_columns:
                    if feature == "frequency[Hz]":
                        continue

                    baseline_value = baseline_means[feature]
                    if baseline_value != 0:
                        pct_change = np.round(
                            (
                                (drug_means_series[feature] - baseline_value)
                                / abs(baseline_value)
                            )
                            * 100,
                            2,
                        )
                    else:
                        pct_change = np.nan
                    changes[feature] = pct_change

                relative_diff_df = pd.concat(
                    [relative_diff_df, pd.DataFrame([changes])]
                )

        relative_diff_df = relative_diff_df.reset_index(drop=True)

        logger.info(
            f"Target concentration changes calculated for {len(drug_names)} drugs"
        )

        return mean_features_df, relative_diff_df

    def get_drug_name_mapping(self, drug_name: str) -> str:
        """
        Get properly formatted drug name for plotting.

        Args:
            drug_name (str): Original drug name

        Returns:
            str: Formatted drug name
        """
        mapping = {
            "nifedipine": "Nifedipine",
            "ca_titration": r"$\rm{Ca^{2+}}$ Titration",
            "e4031": "E-4031",
        }
        return mapping.get(drug_name, drug_name)

    def run_comprehensive_analysis(
        self, target_concentrations: Optional[List[float]] = None
    ) -> Dict:
        """
        Run comprehensive relative comparison analysis at all three levels.

        This method orchestrates the complete analysis pipeline, calculating
        tissue-specific, global average, and target concentration changes.

        Args:
            target_concentrations (Optional[List[float]]): Target concentrations for EC50/IC50 analysis

        Returns:
            Dict: Dictionary containing all analysis results

        Example:
            >>> results = analyzer.run_comprehensive_analysis(
            ...     target_concentrations=[0.03, 1.0, 1.0]
            ... )
        """
        logger.info("Running comprehensive relative comparison analysis...")

        # Prepare drug dataframes
        drug_dfs = {
            "e4031": self.signal_data.e4031_cases,
            "nifedipine": self.signal_data.nifedipine_cases,
            "ca_titration": self.signal_data.ca_titration_cases,
        }

        # Level 1: Tissue-specific changes
        tissue_mean_features, tissue_relative_diffs = (
            self.calculate_tissue_specific_changes(
                self.signal_data.baseline_cases, drug_dfs
            )
        )

        # Level 2: Global average changes
        global_mean_features, global_relative_diffs = (
            self.calculate_global_average_changes(
                self.signal_data.baseline_cases, drug_dfs
            )
        )

        # Level 3: Target concentration changes (if provided)
        target_mean_features = None
        target_relative_diffs = None

        if target_concentrations is not None:
            target_mean_features, target_relative_diffs = (
                self.calculate_target_concentration_changes(
                    self.signal_data.baseline_cases, drug_dfs, target_concentrations
                )
            )

        # Store results
        self.analysis_results = {
            "tissue_specific": {
                "mean_features": tissue_mean_features,
                "relative_differences": tissue_relative_diffs,
            },
            "global_average": {
                "mean_features": global_mean_features,
                "relative_differences": global_relative_diffs,
            },
            "target_concentration": (
                {
                    "mean_features": target_mean_features,
                    "relative_differences": target_relative_diffs,
                    "concentrations": target_concentrations,
                }
                if target_concentrations is not None
                else None
            ),
            "metadata": {
                "feature_columns": self.feature_columns,
                "drug_names": list(drug_dfs.keys()),
                "analysis_timestamp": pd.Timestamp.now(),
            },
        }

        logger.info("Comprehensive relative comparison analysis completed")

        return self.analysis_results

    def save_results(self, output_path: str) -> Dict[str, str]:
        """
        Save analysis results to CSV files.

        Args:
            output_path (str): Directory to save results

        Returns:
            Dict[str, str]: Dictionary mapping result types to file paths
        """
        import os

        os.makedirs(output_path, exist_ok=True)

        saved_files = {}

        # Save tissue-specific results
        if "tissue_specific" in self.analysis_results:
            tissue_results = self.analysis_results["tissue_specific"]
            tissue_results["mean_features"].to_csv(
                os.path.join(output_path, "tissue_specific_mean_features.csv"),
                index=False,
            )
            tissue_results["relative_differences"].to_csv(
                os.path.join(output_path, "tissue_specific_relative_differences.csv"),
                index=False,
            )
            saved_files["tissue_specific"] = output_path

        # Save global average results
        if "global_average" in self.analysis_results:
            global_results = self.analysis_results["global_average"]
            global_results["mean_features"].to_csv(
                os.path.join(output_path, "global_mean_features.csv"), index=False
            )
            global_results["relative_differences"].to_csv(
                os.path.join(output_path, "global_relative_differences.csv"),
                index=False,
            )
            saved_files["global_average"] = output_path

        # Save target concentration results
        if (
            self.analysis_results.get("target_concentration") is not None
            and self.analysis_results["target_concentration"]["mean_features"]
            is not None
        ):
            target_results = self.analysis_results["target_concentration"]
            target_results["mean_features"].to_csv(
                os.path.join(output_path, "target_concentration_mean_features.csv"),
                index=False,
            )
            target_results["relative_differences"].to_csv(
                os.path.join(
                    output_path, "target_concentration_relative_differences.csv"
                ),
                index=False,
            )
            saved_files["target_concentration"] = output_path

        logger.info(f"Analysis results saved to {output_path}")

        return saved_files

    def get_analysis_summary(self) -> Dict:
        """
        Get summary statistics of the analysis results.

        Returns:
            Dict: Summary statistics and metadata
        """
        summary = {
            "total_features": len(self.feature_columns),
            "feature_names": self.feature_columns,
            "analysis_levels": [],
        }

        if "tissue_specific" in self.analysis_results:
            tissue_data = self.analysis_results["tissue_specific"][
                "relative_differences"
            ]
            summary["analysis_levels"].append(
                {
                    "level": "tissue_specific",
                    "total_comparisons": len(tissue_data),
                    "unique_tissues": tissue_data["bct_id"].nunique(),
                    "unique_drugs": tissue_data["drug"].nunique(),
                    "unique_concentrations": tissue_data["concentration[um]"].nunique(),
                }
            )

        if "global_average" in self.analysis_results:
            global_data = self.analysis_results["global_average"][
                "relative_differences"
            ]
            summary["analysis_levels"].append(
                {
                    "level": "global_average",
                    "total_comparisons": len(global_data),
                    "unique_drugs": global_data["drug"].nunique(),
                }
            )

        if (
            self.analysis_results.get("target_concentration") is not None
            and self.analysis_results["target_concentration"]["relative_differences"]
            is not None
        ):
            target_data = self.analysis_results["target_concentration"][
                "relative_differences"
            ]
            summary["analysis_levels"].append(
                {
                    "level": "target_concentration",
                    "total_comparisons": len(target_data),
                    "target_concentrations": self.analysis_results[
                        "target_concentration"
                    ]["concentrations"],
                }
            )

        return summary
