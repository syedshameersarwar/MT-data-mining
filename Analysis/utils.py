from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


@dataclass
class FeatureMapping:
    """
    Class to handle feature name mappings between code and thesis formats.

    This class provides methods to convert between internal feature names used in
    the code and formatted names suitable for thesis presentation. It handles:
    - Basic feature mappings
    - Force-related features
    - Calcium-related features
    - Unit additions
    - Special character formatting
    """

    @staticmethod
    def create_mapping() -> Dict[str, str]:
        """Create mapping between code feature names and thesis-formatted names"""
        return {
            # FP features
            "duration": "FPD [s]",
            "frequency[Hz]": "Frequency [Hz]",
            "local_frequency[Hz]": "Local Frequency [Hz]",
            # Force features
            "force_peak_amplitude": "Force Peak Amplitude [mN]",
            "force_rise_time_0.2_0.8 s": "Force RT 0.2-0.8 [s]",
            "force_rise_time_0.8_max s": "Force RT 0.8-Max [s]",
            "force_decay_time_max_0.8 s": "Force DT Max-0.8 [s]",
            "force_decay_time_0.2_0.8 s": "Force DT 0.2-0.8 [s]",
            "force_width_0.2 s": "Force Width 0.2 [s]",
            "force_width_0.5 s": "Force Width 0.5 [s]",
            "force_width_0.8 s": "Force Width 0.8 [s]",
            # Calcium features
            "calc_peak_amplitude": "Ca$^{2+}$ Peak Amplitude [a.u.]",
            "calc_rise_time_0.2_0.8 s": "Ca$^{2+}$ RT 0.2-0.8 [s]",
            "calc_rise_time_0.8_max s": "Ca$^{2+}$ RT 0.8-Max [s]",
            "calc_decay_time_max_0.8 s": "Ca$^{2+}$ DT Max-0.8 [s]",
            "calc_decay_time_0.2_0.8 s": "Ca$^{2+}$ DT 0.2-0.8 [s]",
            "calc_width_0.2 s": "Ca$^{2+}$ Width 0.2 [s]",
            "calc_width_0.5 s": "Ca$^{2+}$ Width 0.5 [s]",
            "calc_width_0.8 s": "Ca$^{2+}$ Width 0.8 [s]",
        }

    @staticmethod
    def get_thesis_name(feature: str, mapping: Optional[Dict] = None) -> str:
        """
        Convert code feature name to thesis-formatted name.

        Applies formatting rules including:
        - Unit addition based on feature type
        - Special character handling (e.g., Ca2+)
        - Proper capitalization and spacing

        Args:
            feature (str): Internal feature name from code
            mapping (Optional[Dict]): Custom mapping dictionary

        Returns:
            str: Formatted feature name suitable for thesis
        """
        if mapping is None:
            mapping = FeatureMapping.create_mapping()

        if feature in mapping:
            return mapping[feature]

        formatted = feature.replace("_", " ").title()

        if "[" not in formatted:
            if "force" in feature.lower():
                formatted += " [mN]"
            elif "time" in feature.lower() or "duration" in feature.lower():
                formatted += " [s]"
            elif "calc" in feature.lower():
                formatted += " [F/F₀]"

        return formatted


class DataPreprocessor:
    """
    Class for handling data preprocessing operations.

    This class provides methods for cleaning and preprocessing signal data, including:
    - Concentration parsing and unit conversion
    - Local frequency calculation
    - Statistical summary generation
    - Feature selection and filtering

    Attributes:
        FEATURES_TO_KEEP (List[str]): List of features to retain after preprocessing
    """

    FEATURES_TO_KEEP = [
        "duration",
        "force_peak_amplitude",
        "force_width_0.5 s",
        "force_width_0.2 s",
        "force_width_0.8 s",
        "force_rise_time_0.2_0.8 s",
        "force_rise_time_0.8_max s",
        "force_decay_time_max_0.8 s",
        "force_decay_time_0.2_0.8 s",
        "calc_peak_amplitude",
        "calc_width_0.5 s",
        "calc_width_0.2 s",
        "calc_width_0.8 s",
        "calc_rise_time_0.2_0.8 s",
        "calc_rise_time_0.8_max s",
        "calc_decay_time_max_0.8 s",
        "calc_decay_time_0.2_0.8 s",
    ]

    @staticmethod
    def parse_concentration(concentration: str) -> float:
        """
        Parse concentration string to float value in μM.

        Handles various concentration formats and units:
        - mM/mm -> μM (×1000)
        - nM/nm -> μM (÷1000)
        - μM/um -> μM (as is)
        - baseline -> 0
        - Special cases (e.g., "1h" -> 20.0)

        Args:
            concentration (str): Concentration string with units

        Returns:
            float: Concentration value in μM
        """
        # if concentration ends with mM or mm, convert to uM
        if concentration.endswith(("mM", "mm")):
            return float(concentration[:-2]) * 1000
        # if concentration ends with nM or nm, convert to uM
        elif concentration.endswith(("nM", "nm")):
            return float(concentration[:-2]) / 1000
        # if baseline, return 0
        elif "baseline" in concentration:
            return 0
        # if concentration ends with um or Um, return float concentration
        elif concentration.lower().endswith("um"):
            return float(concentration[:-2])
        # if concentration is 1h, return 20.0 (1 hour post-treatment, will be overwritten later in plotting)
        elif concentration == "1h":
            return 20.0  # Special case
        return float(concentration)

    @staticmethod
    def calculate_local_frequency(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate local frequency based on consecutive peak time differences of Na+ spikes. (in Hz)

        Args:
            df (pd.DataFrame): Input DataFrame with peak times

        Returns:
            pd.DataFrame: DataFrame with local frequency calculated
        """
        df["time_diff_prev_peak"] = df["start_time"].diff()
        df["time_diff_next_peak"] = -df["start_time"].diff(-1)
        df["local_frequency[Hz]"] = (
            df["time_diff_prev_peak"] + df["time_diff_next_peak"]
        ) / (2 * (df["time_diff_prev_peak"] * df["time_diff_next_peak"]))

        # Clean up temporary columns
        df = df.drop(columns=["time_diff_prev_peak", "time_diff_next_peak"])
        return df.iloc[1:-1]  # Remove first and last rows (NaN values)

    @staticmethod
    def generate_contractions_count_per_case(case_df: pd.DataFrame) -> Dict:
        """
        Generate summary count of contractions in a case.

        Args:
            case_df (pd.DataFrame): Case data

        Returns:
            Dict: metadata and count of contractions (normal, arrythmic, non-outliers, potential outliers)
        """
        return {
            "drug": case_df.iloc[0]["drug"],
            "bct_id": case_df.iloc[0]["bct_id"],
            "concentration[um]": case_df.iloc[0]["concentration[um]"],
            "frequency[Hz]": case_df.iloc[0]["frequency[Hz]"],
            "events": len(case_df),
            "normal_contraction": len(
                case_df[
                    (case_df["force_arrythmia_prediction"] == 0)
                    | (case_df["calc_arrythmia_prediction"] == 0)
                ]
            ),
            "arrythmic_contractions": len(
                case_df[
                    (case_df["force_arrythmia_prediction"] == 1)
                    | (case_df["calc_arrythmia_prediction"] == 1)
                ]
            ),
            "non-outliers": len(
                case_df[
                    (case_df["force_potential_outlier"] == 0)
                    & (case_df["calc_potential_outlier"] == 0)
                ]
            ),
            "potential_outliers": len(
                case_df[
                    (case_df["force_potential_outlier"] == 1)
                    | (case_df["calc_potential_outlier"] == 1)
                ]
            ),
        }


class SignalData:
    """
    Class to handle signal measurement data and preprocessing.

    This class manages the loading, preprocessing, and organization of multimodal cardiac
    signal data. It handles data for different drug treatments (E-4031, Nifedipine,
    Ca2+ titration) and their corresponding baseline measurements.

    Attributes:
        baseline_cases (pd.DataFrame): Processed baseline measurements
        e4031_cases (pd.DataFrame): Processed E-4031 treatment measurements
        nifedipine_cases (pd.DataFrame): Processed Nifedipine treatment measurements
        ca_titration_cases (pd.DataFrame): Processed Ca2+ titration measurements
        data_path (Path): Path to the data directory
        preprocessor (DataPreprocessor): Instance of DataPreprocessor for data transformation
        features (List[str]): List of feature names extracted from the data
    """

    def __init__(self, data_path: str):
        """
        Initialize SignalData instance.

        Args:
            data_path (str): Path to the features data directory.
        """
        self.baseline_cases = pd.DataFrame()
        self.e4031_cases = pd.DataFrame()
        self.nifedipine_cases = pd.DataFrame()
        self.ca_titration_cases = pd.DataFrame()
        self.data_path = Path(data_path)
        self.preprocessor = DataPreprocessor()

    @staticmethod
    def get_baseline_concentration(drug: str) -> float:
        """
        Get baseline concentration for a given drug.

        Args:
            drug (str): Drug name

        Returns:
            float: Baseline concentration in uM

        Raises:
            ValueError: If drug is not supported
        """
        concentrations = {"e-4031": 0.0, "ca_titration": 4.0, "nifedipine": 0.0}
        if drug not in concentrations:
            raise ValueError(f"Drug {drug} not supported")
        return concentrations[drug]

    @staticmethod
    def get_discarded_concentrations(drug: str) -> List[float]:
        """
        Get discarded concentrations for a given drug.

        Args:
            drug (str): Drug name
        """
        concentrations = {"e-4031": [], "ca_titration": [], "nifedipine": [20]}
        return concentrations[drug]

    def read_all_cases(self, exp_prefix: str = "run1b_e-4031") -> List[str]:
        """
        Read all case files matching the experiment prefix.

        Args:
            exp_prefix (str): Experiment prefix to match

        Returns:
            List[str]: List of file paths
        """
        files = os.listdir(self.data_path)
        return [
            os.path.join(self.data_path, file)
            for file in files
            if re.match(exp_prefix + ".*\.h5", file)
        ]

    def preprocess_case(
        self, case_path: str, all_case_prefix: str = "run1b_"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess a single case file.

        Args:
            case_path (str): Path to case file
            all_case_prefix (str): Prefix to remove from filename

        Returns:
            Tuple[pd.DataFrame, Dict]: Processed DataFrame and contraction record
        """
        df = pd.read_hdf(case_path)

        # Extract metadata from filename
        file_name = Path(case_path).name.replace(all_case_prefix, "")
        if "e-4031" in file_name:
            file_name = file_name.replace("e-4031", "e4031")

        # Parse file name components
        drug, bct_id, conc_part = file_name.split("-")[:3]
        concentration = self.preprocessor.parse_concentration(conc_part.split("_")[1])
        frequency = float(file_name.split("_")[-1][:-3])

        # Add metadata columns
        df = df.assign(
            drug=drug,
            bct_id=bct_id,
        )
        df["concentration[um]"] = concentration
        df["frequency[Hz]"] = frequency

        # Calculate local frequency
        df = self.preprocessor.calculate_local_frequency(df)

        # Generate contraction record
        contraction_count_df = self.preprocessor.generate_contractions_count_per_case(
            df
        )

        # Filter and clean data (remove potential outliers and arrythmic contractions)
        df = df[
            (df["force_potential_outlier"] == 0)
            & (df["calc_potential_outlier"] == 0)
            & (df["force_arrythmia_prediction"] == 0)
            & (df["calc_arrythmia_prediction"] == 0)
        ]

        # Select final columns
        df = df[
            ["drug", "bct_id", "concentration[um]", "frequency[Hz]"]
            + self.preprocessor.FEATURES_TO_KEEP
            + ["local_frequency[Hz]"]
        ]

        return df, contraction_count_df

    def merge_cases_by_drug_and_baseline(
        self, cases_dict: Dict[str, List], discard_concentrations: bool = False
    ):
        """
        Merge cases by drug type and baseline concentration.

        Args:
            cases_dict (Dict[str, List]): Dictionary of cases by drug
        """
        drugs = ["e-4031", "nifedipine", "ca_titration"]
        total_cases = [cases_dict[drug] for drug in drugs]

        for i, drug in enumerate(drugs):
            baseline_conc = self.get_baseline_concentration(drug)
            drug_df = pd.DataFrame()

            for case in total_cases[i]:
                case_df, _ = self.preprocess_case(case)
                if case_df.empty:
                    continue

                case_conc = case_df["concentration[um]"].unique()[0]
                if case_conc == baseline_conc:
                    self.baseline_cases = pd.concat([self.baseline_cases, case_df])
                    # if we have to discard some concentrations, we also add the drug's baseline to selected concentrations
                    if discard_concentrations:
                        drug_df = pd.concat([drug_df, case_df])
                else:
                    drug_df = pd.concat([drug_df, case_df])

            if discard_concentrations:
                drug_df = drug_df[
                    ~drug_df["concentration[um]"].isin(
                        self.get_discarded_concentrations(drug)
                    )
                ]
            setattr(
                self,
                f"{drug.replace('-', '')}_cases",
                drug_df.sort_values(by=["bct_id", "concentration[um]"]),
            )

        self.baseline_cases = self.baseline_cases.sort_values(
            by=["bct_id", "concentration[um]"]
        )
        self.features = self.baseline_cases.columns.tolist()[4:]
