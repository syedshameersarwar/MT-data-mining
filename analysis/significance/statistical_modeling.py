from typing import List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from pymer4.models import Lmer
import rpy2
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import dunnett


class StatisticalAnalyzer:
    """
    A comprehensive statistical analysis class for drug response studies.

    This class performs statistical modeling and hypothesis testing for cardiac drug response
    experiments using both R-based and Python-based statistical packages. It implements
    linear mixed effects models and ANOVA analyses to assess drug treatment effects while
    accounting for tissue-specific variability and repeated measurements.

    Key Statistical Methods:
    1. Linear Mixed Effects Models (LME): Uses R's lmerTest package via rpy2
       - Accounts for tissue-specific random effects
       - Handles nested random effects (concentration within tissue)
       - Provides model singularity diagnostics

    2. Analysis of Variance (ANOVA): Uses Python's statsmodels
       - One-way ANOVA for treatment effects
       - Simplified alternative to mixed effects models

    3. Post-hoc Testing: Dunnett's multiple comparison test
       - Compares all treatment concentrations to baseline/control
       - Controls family-wise error rate

    Model Specifications:
    - Full random effects: response ~ concentration + (1|tissue/concentration)
    - Tissue-only effects: response ~ concentration + (1|tissue)
    - ANOVA model: response ~ C(concentration)

    Attributes:
        utils (rpy2 importr): R utils package for general utilities
        stats_r (rpy2 importr): R stats package for statistical functions
        lmerTest (rpy2 importr): R lmerTest package for mixed effects models
        multcomp (rpy2 importr): R multcomp package for multiple comparisons
        base (rpy2 importr): R base package for core functions
        tidy (rpy2 importr): R tidyverse package for data manipulation
        output_path (Path): Directory path for saving analysis outputs
        selected_concentrations (List[str]): Concentration levels to include in analysis
        tissue_random_only_effects (bool): Whether to use simplified random effects structure

    Example:
        >>> analyzer = StatisticalAnalyzer(
        ...     output_path=Path("./results"),
        ...     selected_concentrations=["0.0", "0.1", "1.0", "10.0"],
        ...     tissue_random_only_effects=False
        ... )
        >>> mixed_results, anova_results = analyzer.analyze_drug_significance(
        ...     drug_df=nifedipine_data,
        ...     drug_name="nifedipine"
        ... )
    """

    def __init__(
        self,
        output_path: Path,
        selected_concentrations: List[str],
        tissue_random_only_effects: bool = False,
    ):
        """
        Initialize the StatisticalAnalyzer with R packages and analysis parameters.

        Sets up the R environment for statistical analysis and configures analysis
        parameters including concentration levels and random effects structure.

        R Package Dependencies:
        - utils: General utility functions
        - stats: Core statistical functions
        - lmerTest: Linear mixed effects modeling with significance tests
        - multcomp: Multiple comparison procedures (Dunnett's test)
        - base: Core R functions
        - tidyverse: Data manipulation and visualization

        Args:
            output_path (Path): Directory where analysis results will be saved.
                Creates subdirectories for data and plots as needed.
            selected_concentrations (List[str]): Concentration values to include
                in statistical analysis. First element should be baseline/control.
                Values should be string representations of numeric concentrations.
            tissue_random_only_effects (bool, optional): Whether to use simplified
                random effects structure. Defaults to False.
                - False: Use nested random effects (1|tissue/concentration)
                - True: Use tissue-only random effects (1|tissue)

        Raises:
            ImportError: If required R packages are not installed
            rpy2.robjects.packages.PackageNotInstalledError: For missing R packages

        Side Effects:
            - Initializes R environment and imports required packages
            - Activates pandas-R data conversion
            - Sets up analysis configuration parameters

        Example:
            >>> # Standard mixed effects analysis
            >>> analyzer = StatisticalAnalyzer(
            ...     output_path=Path("./nifedipine_analysis"),
            ...     selected_concentrations=["0.0", "0.1", "1.0", "10.0"],
            ...     tissue_random_only_effects=False
            ... )

            >>> # Simplified analysis for problematic convergence
            >>> analyzer_simple = StatisticalAnalyzer(
            ...     output_path=Path("./e4031_analysis"),
            ...     selected_concentrations=["0.0", "0.001", "0.01", "0.1"],
            ...     tissue_random_only_effects=True
            ... )
        """

        self.utils = importr("utils")
        self.stats_r = importr("stats")
        self.lmerTest = importr("lmerTest")
        self.multcomp = importr("multcomp")
        self.base = importr("base")
        self.tidy = importr("tidyverse")
        self.output_path = output_path
        self.selected_concentrations = selected_concentrations
        self.tissue_random_only_effects = tissue_random_only_effects

    def _normalize_column(self, df: pd.DataFrame, feature: str):
        """
        Normalize and prepare DataFrame columns for statistical analysis.

        Performs several data preparation steps required for proper statistical modeling:
        1. Standardizes column names for R compatibility
        2. Filters data to selected concentrations only
        3. Converts concentration to categorical variable with proper ordering
        4. Ensures proper data types for statistical analysis

        Column Transformations:
        - "concentration[um]" → "concentration"
        - Feature names: Remove spaces and special characters (e.g., "[Hz]")
        - Concentration: Convert to categorical with baseline as reference level

        Args:
            df (pd.DataFrame): Input DataFrame containing experimental data
            feature (str): Name of the feature column to analyze

        Returns:
            pd.DataFrame: Normalized DataFrame ready for statistical analysis with:
                - Standardized column names
                - Filtered to selected concentrations
                - Categorical concentration variable
                - Proper data types for modeling

        Side Effects:
            - Modifies column names in-place
            - Filters rows based on selected_concentrations
            - Converts data types for statistical compatibility

        Example:
            >>> df_orig = pd.DataFrame({
            ...     'concentration[um]': [0.0, 0.1, 1.0, 10.0, 100.0],
            ...     'force_peak_amplitude': [1.2, 1.1, 0.9, 0.7, 0.5],
            ...     'bct_id': ['tissue1'] * 5
            ... })
            >>> analyzer.selected_concentrations = ['0.0', '0.1', '1.0', '10.0']
            >>> df_norm = analyzer._normalize_column(df_orig, 'force_peak_amplitude')
            >>> df_norm.columns.tolist()
            ['concentration', 'force_peak_amplitude', 'bct_id']
            >>> df_norm['concentration'].dtype
            CategoricalDtype(categories=['0.0', '0.1', '1.0', '10.0'], ordered=False)
        """
        df = df.rename(columns={"concentration[um]": "concentration"})
        df = df.rename(columns={feature: feature.replace(" ", "_").replace("[Hz]", "")})
        df = df[
            df["concentration"].isin([float(c) for c in self.selected_concentrations])
        ]
        df["concentration"] = df["concentration"].astype(str)
        df["concentration"] = pd.Categorical(
            df["concentration"], categories=self.selected_concentrations
        )
        return df

    def run_mixed_model_in_r_and_python(
        self,
        df: pd.DataFrame,
        feature: str,
        tissue_random_only_effects=False,
    ):
        """
        Run comprehensive linear mixed effects model analysis using both R and Python.

        Performs mixed effects modeling using R's lmerTest package for statistical
        inference and Python's pymer4 package for additional diagnostics. This
        dual approach leverages the strengths of both implementations:

        R Implementation (lmerTest):
        - Robust statistical inference with Satterthwaite degrees of freedom
        - Reliable ANOVA F-tests for fixed effects
        - Comprehensive post-hoc testing with multcomp package

        Python Implementation (pymer4):
        - Easy extraction of fitted values and residuals
        - Convenient access to model diagnostics
        - Integration with pandas DataFrames

        Model Specifications:
        1. Full random effects: feature ~ concentration + (1|tissue/concentration)
           - Accounts for tissue-specific intercepts
           - Accounts for tissue-specific concentration effects
           - Used when sufficient data available

        2. Tissue-only effects: feature ~ concentration + (1|tissue)
           - Accounts for tissue-specific intercepts only
           - Used when convergence issues occur with full model

        Post-hoc Testing:
        - Dunnett's test: Compares all concentrations to baseline/control
        - Controls family-wise error rate for multiple comparisons
        - Only performed if overall ANOVA is significant (p < 0.05)

        Args:
            df (pd.DataFrame): Experimental data containing measurements
            feature (str): Name of the response variable to analyze
            tissue_random_only_effects (bool, optional): Whether to use simplified
                random effects structure. Defaults to False.

        Returns:
            Dict: Comprehensive analysis results containing:
                - model (Lmer): Fitted pymer4 model object
                - is_singular (bool): Whether model fit is singular (convergence issues)
                - anova_p_value (float): Overall F-test p-value for concentration effect
                - dunnett_results (pd.DataFrame or None): Post-hoc test results if significant
                - fitted_values (pd.Series): Model predicted values
                - residuals (pd.Series): Model residuals
                - variance_components (Dict): Random effects variance estimates

        Raises:
            RuntimeError: If model fitting fails
            rpy2.rinterface_lib.embedded.RRuntimeError: For R-specific errors

        Side Effects:
            - Normalizes input DataFrame columns
            - Fits statistical models in both R and Python
            - May print convergence warnings

        Example:
            >>> results = analyzer.run_mixed_model_in_r_and_python(
            ...     df=drug_data,
            ...     feature="force_peak_amplitude",
            ...     tissue_random_only_effects=False
            ... )
            >>> print(f"Overall p-value: {results['anova_p_value']:.3f}")
            >>> print(f"Model singular: {results['is_singular']}")
            >>> if results['dunnett_results'] is not None:
            ...     print(results['dunnett_results'])
        """
        df = self._normalize_column(df, feature)
        if not tissue_random_only_effects:
            model_str = f"{feature.replace(' ', '_').replace('[Hz]', '')} ~ concentration + (1|bct_id/concentration)"
        else:
            model_str = f"{feature.replace(' ', '_').replace('[Hz]', '')} ~ concentration + (1|bct_id)"
        with (ro.default_converter + pandas2ri.converter).context():
            lmer_test_model_r = self.lmerTest.lmer(model_str, data=df)
            anova_r = self.stats_r.anova(lmer_test_model_r)
            anova_py_df = ro.conversion.get_conversion().rpy2py(anova_r)
        _lmer_model_py = Lmer(model_str, data=df)
        _lmer_model_py.fit(factors={"concentration": self.selected_concentrations})
        anova_p_val = anova_py_df["Pr(>F)"][0]

        post_hoc = self.multcomp.glht(
            lmer_test_model_r, linfct=self.multcomp.mcp(concentration="Dunnett")
        )
        post_hoc_summary_r = self.base.summary(post_hoc, complete=True)

        indexes = []
        for line in (
            str(post_hoc_summary_r).split("Linear Hypotheses:\n")[1].split("\n")[1:]
        ):
            line = line.strip()
            if "==" in line:
                concentrations = line.split("==")[0].strip()
                conc1 = concentrations.split("-")[0].strip()
                conc2 = concentrations.split("-")[1].strip()
                indexes.append(f"concentration{conc2}-concentration{conc1}")
        coefficients = post_hoc_summary_r.rx2("test").rx2("coefficients")
        post_hoc_p_values = post_hoc_summary_r.rx2("test").rx2("pvalues")

        dunnett_results = pd.DataFrame(
            {"Contrast": indexes, "Estimate": coefficients, "P-val": post_hoc_p_values}
        )

        return {
            "model": _lmer_model_py,
            "is_singular": len(_lmer_model_py.warnings) > 0
            and "singular" in _lmer_model_py.warnings[0],
            "anova_p_value": anova_p_val,
            "dunnett_results": None if anova_p_val > 0.05 else dunnett_results,
            "fitted_values": _lmer_model_py.fits,
            "residuals": _lmer_model_py.residuals,
            "variance_components": self.extract_variance_components(_lmer_model_py),
        }

    def run_anova_analysis(self, df: pd.DataFrame, feature: str):
        """
        Perform one-way ANOVA analysis with Dunnett's post-hoc testing.

        Conducts a simplified statistical analysis using one-way ANOVA instead of
        mixed effects models. This approach is useful for comparison purposes or
        when mixed effects models encounter convergence issues.

        Analysis Steps:
        1. One-way ANOVA: Tests for overall concentration effect
        2. Dunnett's Test: Multiple comparisons against baseline (if ANOVA significant)
        3. Model Diagnostics: Extract fitted values and residuals

        Model Specification:
        - ANOVA: response ~ C(concentration)
        - Treats concentration as categorical factor
        - Ignores repeated measures structure (no random effects)

        Post-hoc Testing:
        - Uses scipy.stats.dunnett for multiple comparisons
        - Compares each treatment concentration to baseline/control
        - Only performed if overall ANOVA F-test is significant

        Args:
            df (pd.DataFrame): Experimental data with concentration and feature columns
            feature (str): Name of the response variable to analyze

        Returns:
            Dict: Analysis results containing:
                - model (statsmodels RegressionResults): Fitted ANOVA model
                - is_singular (bool): Always False for ANOVA models
                - anova_p_value (float): F-test p-value for concentration effect
                - dunnett_results (pd.DataFrame or None): Post-hoc comparisons if significant
                - fitted_values (pd.Series): Model predicted values (group means)
                - residuals (pd.Series): Model residuals
                - variance_components (Dict): Residual variance estimate

        Raises:
            ValueError: If feature column contains non-numeric data
            KeyError: If required columns are missing from DataFrame

        Side Effects:
            - Normalizes DataFrame column names
            - Filters data to selected concentrations
            - Fits statistical model using statsmodels

        Example:
            >>> anova_results = analyzer.run_anova_analysis(
            ...     df=drug_data,
            ...     feature="calc_peak_amplitude"
            ... )
            >>> print(f"ANOVA p-value: {anova_results['anova_p_value']:.3f}")
            >>> if anova_results['dunnett_results'] is not None:
            ...     print("Significant post-hoc comparisons found")
        """
        baseline = float(self.selected_concentrations[0])
        df = df.rename(columns={"concentration[um]": "concentration"})
        df = df.rename(
            columns={
                feature: feature.replace(" ", "_").replace("[Hz]", "").replace(".", "")
            }
        )
        df = df[
            df["concentration"].isin([float(c) for c in self.selected_concentrations])
        ]

        # Fit one-way ANOVA using statsmodels
        model_str = f"{feature.replace(' ', '_').replace('[Hz]', '').replace('.', '')} ~ C(concentration)"
        model = ols(model_str, data=df).fit()
        anova_table = anova_lm(model, typ=2)
        anova_p_value = anova_table["PR(>F)"][0]

        # Run Dunnett's test if ANOVA is significant
        posthoc_results = None
        if anova_p_value < 0.05:
            # Get control group data (baseline)
            feature_name = (
                feature.replace(" ", "_").replace("[Hz]", "").replace(".", "")
            )
            control_data = df[df["concentration"] == baseline][feature_name].values
            # Prepare data for each non-baseline concentration
            other_concentrations = sorted(
                [c for c in df["concentration"].unique() if c != baseline]
            )
            treatment_groups = []
            contrasts = []

            for conc in other_concentrations:
                treatment_data = df[df["concentration"] == conc][feature_name].values
                treatment_groups.append(treatment_data)
                contrasts.append(f"concentration{conc}-concentration{baseline}")

            # Run Dunnett's test
            dunnett_results = dunnett(
                *treatment_groups,
                control=control_data,
                random_state=42,
            )

            # Create DataFrame with results
            baseline_comparisons = []
            for i, p_val in enumerate(dunnett_results.pvalue):
                baseline_comparisons.append({"Contrast": contrasts[i], "P-val": p_val})
            posthoc_results = pd.DataFrame(baseline_comparisons)

        return {
            "model": model,
            "is_singular": False,
            "anova_p_value": anova_p_value,
            "dunnett_results": posthoc_results,
            "fitted_values": model.fittedvalues,
            "residuals": model.resid,
            "variance_components": self.extract_variance_components(model),
        }

    def extract_variance_components(self, model) -> Dict[str, float]:
        """
        Extract variance components from fitted statistical models.

        Retrieves variance estimates for different sources of variation in the model.
        For mixed effects models, this includes random effects variances. For ANOVA
        models, this includes only residual variance.

        Variance Components:
        - σ²_α (sigma_alpha): Tissue-specific random intercept variance
        - σ²_β (sigma_beta): Concentration×tissue interaction variance (if nested RE)
        - σ²_ε (sigma_epsilon): Residual/error variance

        Model Type Handling:
        - Mixed Effects (pymer4.Lmer): Extracts all random effects variances
        - ANOVA (statsmodels): Only residual variance available

        Args:
            model: Fitted statistical model object
                - pymer4.Lmer: Mixed effects model with ranef_var attribute
                - statsmodels.RegressionResults: ANOVA model with mse_resid attribute

        Returns:
            Dict[str, float]: Dictionary of variance components:
                - For mixed effects with nested RE: sigma_alpha, sigma_beta, sigma_epsilon
                - For mixed effects with simple RE: sigma_alpha, sigma_epsilon
                - For ANOVA models: sigma_epsilon only

        Example:
            >>> # Mixed effects model with nested random effects
            >>> variance_comp = analyzer.extract_variance_components(lmer_model)
            >>> print(variance_comp)
            {'sigma_alpha': 0.15, 'sigma_beta': 0.08, 'sigma_epsilon': 0.12}

            >>> # ANOVA model
            >>> variance_comp = analyzer.extract_variance_components(anova_model)
            >>> print(variance_comp)
            {'sigma_epsilon': 0.18}
        """
        if hasattr(model, "ranef_var"):
            variance_estimates = model.ranef_var.iloc[:, -1]
            components = {}
            if len(variance_estimates) == 3:
                components.update(
                    {
                        "sigma_beta": variance_estimates.iloc[0],
                        "sigma_alpha": variance_estimates.iloc[1],
                        "sigma_epsilon": variance_estimates.iloc[2],
                    }
                )
            elif len(variance_estimates) == 2:
                components.update(
                    {
                        "sigma_alpha": variance_estimates.iloc[0],
                        "sigma_epsilon": variance_estimates.iloc[1],
                    }
                )
            return components
        return {
            "sigma_epsilon": model.mse_resid,
        }

    def create_frequency_table(self, df: pd.DataFrame):
        """
        Generate frequency table showing measurement counts across conditions.

        Creates a cross-tabulation table displaying the number of measurements
        for each tissue-concentration combination. This table is useful for:
        1. Assessing experimental design balance
        2. Identifying missing data patterns
        3. Verifying data completeness before analysis

        Table Structure:
        - Rows: Tissue identifiers (bct_id)
        - Columns: Concentration levels (including baseline)
        - Values: Count of measurements per combination
        - Margins: Row and column totals

        Column Ordering:
        - Baseline concentration first
        - Other concentrations in ascending order
        - "Total" column last

        Args:
            df (pd.DataFrame): Experimental data with concentration and tissue columns

        Returns:
            pd.DataFrame: Cross-tabulation table with:
                - Index: Tissue identifiers (bct_id)
                - Columns: Concentration levels + "Total"
                - Values: Measurement counts
                - Last row: Column totals

        Side Effects:
            - Filters data to selected concentrations only
            - Creates a copy of input DataFrame (no modification)

        Example:
            >>> freq_table = analyzer.create_frequency_table(drug_data)
            >>> print(freq_table)
                     0.0  0.1  1.0  10.0  Total
            bct_id
            tissue1   45   42   38    41    166
            tissue2   38   40   35    37    150
            tissue3   41   38   33    35    147
            Total    124  120  106   113    463
        """
        df = df.copy()[
            df["concentration[um]"].isin(
                [float(c) for c in self.selected_concentrations]
            )
        ]
        baseline = self.selected_concentrations[0]

        # Create pivot table
        freq_table = pd.crosstab(
            index=df["bct_id"],
            columns=df["concentration[um]"],
            margins=True,  # Add row and column totals
            margins_name="Total",
        )
        # Sort columns with baseline first, then other concentrations
        column_order = (
            [float(baseline)]
            + sorted([c for c in freq_table.columns if c != baseline and c != "Total"])
            + ["Total"]
        )
        freq_table = freq_table[column_order]
        return freq_table

    def analyze_drug_significance(
        self,
        drug_df: pd.DataFrame,
        drug_name: str,
        feature_subset: List[str] = None,
        exclude_subset: bool = False,
    ):
        """
        Perform comprehensive statistical significance analysis for drug effects.

        Conducts complete statistical analysis workflow for a drug treatment study:
        1. Feature selection based on subset parameters
        2. Frequency table generation for experimental design verification
        3. Mixed effects modeling for each selected feature
        4. ANOVA analysis for comparison purposes
        5. Results compilation and storage

        Analysis Workflow:
        - For each feature: Fits both mixed effects and ANOVA models
        - Extracts statistical measures: p-values, effect sizes, diagnostics
        - Performs post-hoc testing when overall effects are significant
        - Saves frequency table showing experimental design balance

        Feature Selection Logic:
        - If feature_subset is None: Analyze all available features
        - If exclude_subset=False: Analyze only features in subset
        - If exclude_subset=True: Analyze all features except those in subset

        Args:
            drug_df (pd.DataFrame): Complete drug response dataset with columns:
                - First 4 columns: Metadata (drug, bct_id, concentration, frequency)
                - Remaining columns: Feature measurements to analyze
            drug_name (str): Name of drug treatment for output organization
            feature_subset (List[str], optional): Specific features to include/exclude.
                If None, uses all available features.
            exclude_subset (bool, optional): Whether to exclude rather than include
                features in feature_subset. Defaults to False.

        Returns:
            Tuple[Dict, Dict]: Two dictionaries containing analysis results:
                - mixed_model_results: Results from linear mixed effects models
                - anova_results: Results from one-way ANOVA models
                Both dicts have feature names as keys and result dictionaries as values.

        Side Effects:
            - Creates output directory structure under self.output_path
            - Saves frequency table as CSV file
            - Prints progress information during analysis

        Directory Structure Created:
            output_path/
            └── data/
                └── frequency_table.csv

        Example:
            >>> # Analyze subset of features
            >>> mixed_res, anova_res = analyzer.analyze_drug_significance(
            ...     drug_df=nifedipine_data,
            ...     drug_name="nifedipine",
            ...     feature_subset=["duration", "force_peak_amplitude"],
            ...     exclude_subset=False
            ... )

            >>> # Analyze all features except specified ones
            >>> mixed_res, anova_res = analyzer.analyze_drug_significance(
            ...     drug_df=e4031_data,
            ...     drug_name="e-4031",
            ...     feature_subset=["outlier_features"],
            ...     exclude_subset=True
            ... )
        """
        mixed_model_results = {}
        anova_results = {}

        all_features = drug_df.columns[4:].tolist()
        features = all_features
        if feature_subset is not None:
            if exclude_subset:
                features = [f for f in all_features if f not in feature_subset]
            else:
                features = [f for f in all_features if f in feature_subset]

        frequency_table = self.create_frequency_table(drug_df)
        (self.output_path / "data").mkdir(parents=True, exist_ok=True)
        frequency_table.to_csv(self.output_path / f"data/frequency_table.csv")

        for feature in features:
            mixed_model_results[feature] = self.run_mixed_model_in_r_and_python(
                drug_df, feature, self.tissue_random_only_effects
            )
            anova_results[feature] = self.run_anova_analysis(drug_df, feature)

        return mixed_model_results, anova_results
