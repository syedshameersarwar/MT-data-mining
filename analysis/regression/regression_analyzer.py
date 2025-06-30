"""
Regression Analysis Module for Cardiac Signal Feature Analysis

This module provides comprehensive regression analysis capabilities using GLMM Lasso
regression to examine feature relationships under different drug treatments. The analysis
performs feature-wise regression with tissue and concentration random effects, using
Lasso shrinkage for feature selection.

Scientific Background:
    GLMM Lasso regression reveals:
    - Feature interdependencies and relationships
    - Drug-specific changes in feature correlations
    - Tissue and concentration level random effects
    - Sparse feature selection through Lasso shrinkage

Key Analysis Components:
    1. Data preparation and standardization
    2. GLMM Lasso regression with cross-validation
    3. Coefficient matrix generation
    4. Feature relationship comparison across conditions
    5. Statistical significance assessment

Pipeline Architecture:
    Signal Data → Data Preparation → R Script Execution → Coefficient Analysis → Results Export

Supported Analysis Types:
    - Feature-wise regression with random effects
    - Cross-validated Lasso parameter selection
    - Coefficient pattern comparison
    - Drug treatment effect analysis

Drug Treatments Analyzed:
    - Baseline: Control conditions
    - E-4031: hERG potassium channel blocker
    - Nifedipine: L-type calcium channel blocker
    - Ca²⁺ Titration: Calcium concentration modulation

Output Products:
    - Coefficient matrices for each condition
    - Feature relationship patterns
    - Cross-validation results
    - Statistical summaries

Authors: Cardiac Electrophysiology Research Team
Version: 1.0
Dependencies: pandas, numpy, subprocess, pathlib
"""

import pandas as pd
import numpy as np
import subprocess
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils import SignalData, FeatureMapping

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegressionAnalyzer:
    """
    Class for performing GLMM Lasso regression analysis on cardiac signal features.

    This class handles the preparation of data for regression analysis, execution
    of R scripts for GLMM Lasso regression, and processing of coefficient matrices.
    It performs feature-wise regression with tissue and concentration random effects.

    Attributes:
        signal_data (SignalData): Processed signal data object
        r_script_path (Path): Path to the R script for GLMM Lasso
        output_path (Path): Directory for saving results
        analysis_results (Dict): Dictionary storing all analysis results
    """

    def __init__(
        self,
        signal_data: SignalData,
        r_script_path: str = "glmmLasso.R",
        output_path: str = "./data",
    ):
        """
        Initialize RegressionAnalyzer with signal data and configuration.

        Args:
            signal_data (SignalData): Processed signal data object
            r_script_path (str): Path to the R script for GLMM Lasso
            output_path (str): Directory for saving results
        """
        self.signal_data = signal_data
        self.r_script_path = Path(r_script_path)
        self.output_path = Path(output_path)
        self.analysis_results = {}

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RegressionAnalyzer with R script: {r_script_path}")

    def get_drug_baseline_and_discarded_concentrations(
        self, drug: str
    ) -> Tuple[float, List[float]]:
        """
        Get baseline and discarded concentrations for a given drug.

        Args:
            drug (str): Drug name

        Returns:
            Tuple[float, List[float]]: Baseline concentration and list of discarded concentrations
        """
        if drug == "e-4031" or drug == "e4031":
            return 0.0, []
        elif drug == "ca_titration":
            return 4.0, [0.1, 0.2, 0.8]
        elif drug == "nifedipine":
            return 0.0, [1.0, 10.0]
        else:
            raise ValueError(f"Drug {drug} not supported")

    def prepare_data_for_regression(self, drug_name: str) -> pd.DataFrame:
        """
        Prepare data for regression analysis by filtering and organizing.

        Args:
            drug_name (str): Name of the drug treatment

        Returns:
            pd.DataFrame: Prepared data for regression analysis
        """
        logger.info(f"Preparing data for {drug_name} regression analysis...")
        # Select appropriate data based on drug
        if drug_name == "baseline":
            data = self.signal_data.baseline_cases.copy()
        elif drug_name == "e-4031" or drug_name == "e4031":
            data = self.signal_data.e4031_cases.copy()
        elif drug_name == "nifedipine":
            data = self.signal_data.nifedipine_cases.copy()
        elif drug_name == "ca_titration":
            data = self.signal_data.ca_titration_cases.copy()
        else:
            raise ValueError(f"Unknown drug: {drug_name}")

        discarded_conc = None
        if drug_name != "baseline":
            _, discarded_conc = self.get_drug_baseline_and_discarded_concentrations(
                drug_name
            )

        # Filter out discarded concentrations (concentrations are discarded which dont have both combination of tissues)
        if discarded_conc:
            data = data[~data["concentration[um]"].isin(discarded_conc)]

        # Ensure we have data
        if data.empty:
            logger.warning(f"No data available for {drug_name}")
            return pd.DataFrame()

        # Select relevant columns for regression
        feature_columns = self.signal_data.features

        # Create regression data with drug column as first column
        regression_data = data[
            ["bct_id", "concentration[um]", "frequency[Hz]"] + feature_columns
        ].copy()

        # Add drug column as the first column
        regression_data.insert(0, "drug", drug_name)

        # Rename columns for R compatibility
        regression_data = regression_data.rename(
            columns={"bct_id": "bct_id", "concentration[um]": "concentration.um."}
        )

        logger.info(
            f"Prepared {len(regression_data)} samples for {drug_name} regression"
        )

        return regression_data

    def save_data_for_r_script(self, data: pd.DataFrame, drug_name: str) -> str:
        """
        Save prepared data to CSV file for R script processing.

        Args:
            data (pd.DataFrame): Prepared data for regression
            drug_name (str): Name of the drug treatment

        Returns:
            str: Path to the saved CSV file
        """
        if data.empty:
            logger.warning(f"No data to save for {drug_name}")
            return ""

        # Save to CSV
        csv_path = self.output_path / f"{drug_name}.csv"
        data.to_csv(csv_path, index=False)

        logger.info(f"Saved {drug_name} data to {csv_path}")

        return str(csv_path)

    def run_r_script(self) -> bool:
        """
        Execute the R script for GLMM Lasso regression analysis.

        The R script (glmmLasso.R) is designed to run for all drugs (baseline, e4031,
        nifedipine, ca_titration) sequentially. It reads the CSV files prepared by
        the Python code and generates coefficient matrices for each drug.

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Running R script for GLMM Lasso regression analysis...")
        logger.info("This may take 10-30 minutes depending on data size...")

        try:
            # Find Rscript path
            import subprocess

            try:
                # Try to find Rscript in PATH
                result = subprocess.run(
                    ["which", "Rscript"], capture_output=True, text=True, check=True
                )
                rscript_path = result.stdout.strip()
                logger.info(f"Found Rscript at: {rscript_path}")
            except subprocess.CalledProcessError:
                # Fallback to common locations
                common_paths = ["/usr/bin/Rscript", "/usr/local/bin/Rscript", "Rscript"]
                rscript_path = None
                for path in common_paths:
                    if subprocess.run(["test", "-f", path], shell=True).returncode == 0:
                        rscript_path = path
                        logger.info(f"Found Rscript at: {rscript_path}")
                        break

                if rscript_path is None:
                    logger.error("Rscript not found. Please ensure R is installed.")
                    return False

            # Copy R script to output directory
            r_script_name = self.r_script_path.name
            output_r_script = self.output_path / r_script_name

            logger.info(f"Source R script: {self.r_script_path}")
            logger.info(f"Target R script: {output_r_script}")

            # Always copy the R script to ensure it's up to date
            import shutil

            shutil.copy2(self.r_script_path, output_r_script)
            logger.info(f"Copied R script to: {output_r_script}")

            # Verify the file was copied
            if output_r_script.exists():
                logger.info(
                    f"R script successfully copied. Size: {output_r_script.stat().st_size} bytes"
                )
            else:
                logger.error(f"Failed to copy R script to {output_r_script}")
                return False

            # Run R script from the output directory where CSV files are located
            logger.info(f"Executing: {rscript_path} --vanilla {r_script_name}")
            logger.info(f"Working directory: {self.output_path}")

            # Run with real-time output
            result = subprocess.check_call(
                [rscript_path, "--vanilla", r_script_name],
                cwd=self.output_path,
            )

            logger.info("R script completed successfully for all drugs")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"R script failed with exit code {e.returncode}")
            return False
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error running R script: {e}")
            return False

    def read_coefficient_matrix(self, drug_name: str) -> pd.DataFrame:
        """
        Read coefficient matrix from R script output.

        Args:
            drug_name (str): Name of the drug treatment

        Returns:
            pd.DataFrame: Coefficient matrix with formatted feature names
        """
        logger.info(f"Reading coefficient matrix for {drug_name}...")

        # Path to the coefficient matrix file
        matrix_path = self.output_path / f"{drug_name}_weight_matrix.csv"

        if not matrix_path.exists():
            logger.error(f"Coefficient matrix file not found: {matrix_path}")
            return pd.DataFrame()

        try:
            # Read CSV, using first row as headers, and first column as index
            df = pd.read_csv(matrix_path, header=0, index_col=0)

            # Format column names
            new_columns = [
                col.replace(".Hz.", "[Hz]").replace(".s", " s") for col in df.columns
            ]
            df.columns = new_columns

            # Format index names to match columns
            df.index = new_columns

            # Make sure index name matches column names
            df.index.name = None

            logger.info(f"Successfully read coefficient matrix for {drug_name}")

            return df

        except Exception as e:
            logger.error(f"Error reading coefficient matrix for {drug_name}: {e}")
            return pd.DataFrame()

    def create_frequency_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a frequency table showing counts of measurements for each tissue-treatment combination.

        Args:
            data (pd.DataFrame): Input dataframe containing 'bct_id' and 'concentration.um.' columns

        Returns:
            pd.DataFrame: Frequency table with tissues as rows and treatments as columns
        """
        # Create pivot table
        freq_table = pd.crosstab(
            index=data["bct_id"],
            columns=data["concentration.um."],
            margins=True,  # Add row and column totals
            margins_name="Total",
        )

        return freq_table

    def run_regression_analysis(self, drugs: List[str] = None) -> Dict:
        """
        Run complete regression analysis for all specified drugs.

        The R script (glmmLasso.R) is designed to run for all drugs sequentially,
        so this method prepares data for all drugs and then executes the R script once.

        Args:
            drugs (List[str]): List of drugs to analyze. If None, analyzes all drugs.

        Returns:
            Dict: Dictionary containing all analysis results

        Example:
            >>> results = analyzer.run_regression_analysis(['baseline', 'e4031', 'nifedipine'])
        """
        logger.info("Starting complete regression analysis...")

        if drugs is None:
            drugs = ["baseline", "e4031", "nifedipine", "ca_titration"]

        analysis_results = {}

        # Step 1: Prepare data for all drugs
        logger.info("Step 1: Preparing data for all drugs...")
        for drug in drugs:
            logger.info(f"Preparing data for {drug}...")

            # Prepare data
            data = self.prepare_data_for_regression(drug)
            if data.empty:
                logger.warning(f"Skipping {drug} due to empty data")
                continue

            # Save data for R script
            csv_path = self.save_data_for_r_script(data, drug)
            if not csv_path:
                continue

            # Create frequency table
            freq_table = self.create_frequency_table(data)
            freq_table_path = self.output_path / f"{drug}_frequency_table.csv"
            freq_table.to_csv(freq_table_path, index=True)

            # Store initial results
            analysis_results[drug] = {
                "data": data,
                "frequency_table": freq_table,
                "csv_path": csv_path,
                "freq_table_path": str(freq_table_path),
            }

            logger.info(f"Data preparation completed for {drug}")

        # Step 2: Run R script once for all drugs
        logger.info("Step 2: Running R script for all drugs...")
        r_success = self.run_r_script()
        if not r_success:
            logger.error("R script failed, cannot proceed with coefficient reading")
            return analysis_results

        # Step 3: Read coefficient matrices for all drugs
        logger.info("Step 3: Reading coefficient matrices...")
        for drug in drugs:
            if drug in analysis_results:
                # Read coefficient matrix
                coef_matrix = self.read_coefficient_matrix(drug)
                if not coef_matrix.empty:
                    analysis_results[drug]["coefficient_matrix"] = coef_matrix
                    analysis_results[drug]["matrix_path"] = str(
                        self.output_path / f"{drug}_weight_matrix.csv"
                    )
                    logger.info(f"Coefficient matrix read successfully for {drug}")
                else:
                    logger.error(f"Failed to read coefficient matrix for {drug}")

        self.analysis_results = analysis_results

        logger.info(f"Regression analysis completed for {len(analysis_results)} drugs")

        return analysis_results

    def get_analysis_summary(self) -> Dict:
        """
        Get summary statistics of the regression analysis results.

        Returns:
            Dict: Summary statistics and metadata
        """
        summary = {
            "total_drugs": len(self.analysis_results),
            "drugs_analyzed": list(self.analysis_results.keys()),
            "total_features": (
                len(self.signal_data.features) if self.signal_data.features else 0
            ),
            "feature_names": self.signal_data.features,
            "analysis_details": {},
        }

        for drug, results in self.analysis_results.items():
            data = results["data"]

            # Handle case where coefficient_matrix might not exist yet
            coef_matrix = results.get("coefficient_matrix", pd.DataFrame())

            summary["analysis_details"][drug] = {
                "samples": len(data),
                "tissues": data["bct_id"].nunique(),
                "concentrations": data["concentration.um."].nunique(),
                "matrix_shape": coef_matrix.shape if not coef_matrix.empty else (0, 0),
                "non_zero_coefficients": (
                    (coef_matrix != 0).sum().sum() if not coef_matrix.empty else 0
                ),
            }

        return summary

    def save_analysis_results(self) -> Dict[str, str]:
        """
        Save all analysis results to files.

        Returns:
            Dict[str, str]: Dictionary mapping result types to file paths
        """
        saved_files = {}

        for drug, results in self.analysis_results.items():
            # Save data
            data_path = self.output_path / f"{drug}_data.csv"
            results["data"].to_csv(data_path, index=False)
            saved_files[f"{drug}_data"] = str(data_path)

            # Save frequency table
            freq_path = self.output_path / f"{drug}_frequency_table.csv"
            results["frequency_table"].to_csv(freq_path, index=True)
            saved_files[f"{drug}_frequency_table"] = str(freq_path)

            # Save coefficient matrix (if it exists)
            coef_matrix = results.get("coefficient_matrix", pd.DataFrame())
            if not coef_matrix.empty:
                coef_path = self.output_path / f"{drug}_coefficient_matrix.csv"
                coef_matrix.to_csv(coef_path, index=True)
                saved_files[f"{drug}_coefficient_matrix"] = str(coef_path)

        logger.info(f"Saved analysis results to {self.output_path}")

        return saved_files
