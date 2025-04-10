import os
import json
import pandas as pd
import numpy as np
import joblib
import tsfel
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common import CommonFeatureExtractor
from fp import FieldPotentialFeatureExtractor


FORCE_SIGNAL_TYPE = "force"
CALCIUM_SIGNAL_TYPE = "calcium"
FIELD_POTENTIAL_SIGNAL_TYPE = "field_potential"


def init_arrythmia_classifiers(
    force_classifier_path, calcium_classifier_path, raw_data_dir
):
    force_classifier = joblib.load(force_classifier_path)
    calcium_classifier = joblib.load(calcium_classifier_path)

    statistical_cfg_file = tsfel.get_features_by_domain("statistical")
    statistical_cfg_file["statistical"]["ECDF Percentile"]["parameters"][
        "percentile"
    ] = [
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        0.9,
    ]

    force_classifier_window_size = float(
        force_classifier_path.split("_")[-1].replace("s.joblib", "")
    )
    calcium_classifier_window_size = float(
        calcium_classifier_path.split("_")[-1].replace("s.joblib", "")
    )

    def _get_arrythmia_classification(case, df, signal_type, peak_time_idx):
        if signal_type == FORCE_SIGNAL_TYPE:
            classifier = force_classifier
            window_size = force_classifier_window_size
            raw_data_type = "force"
        elif signal_type == CALCIUM_SIGNAL_TYPE:
            classifier = calcium_classifier
            window_size = calcium_classifier_window_size
            raw_data_type = "calc"
        signal = df[raw_data_type]
        if (peak_time_idx - window_size) < signal.index[0] or (
            peak_time_idx + window_size
        ) > signal.index[-1]:
            return -1
        target_signal = signal.loc[
            peak_time_idx - window_size : peak_time_idx + window_size
        ]
        raw_data_df = pd.read_hdf(os.path.join(raw_data_dir, f"{case}__raw_data.hdf"))
        raw_sampling_rate = 1 / (
            raw_data_df[raw_data_type].index[1] - raw_data_df[raw_data_type].index[0]
        )
        preprocessed_sampling_rate = 1 / (signal.index[1] - signal.index[0])
        downsample_factor = preprocessed_sampling_rate // raw_sampling_rate
        target_signal_downsampled = target_signal.iloc[::downsample_factor]
        features = tsfel.time_series_features_extractor(
            statistical_cfg_file,
            target_signal_downsampled.values,
            raw_sampling_rate,
            verbose=0,
        )
        # remove the 0_ prefix from the column names
        col_name = [
            col.replace("0_", "").replace(" ", "_").lower()
            for col in features.columns.to_list()
        ]
        features.columns = col_name
        predicted_class = classifier.predict(features)[0]
        return predicted_class

    return _get_arrythmia_classification


def plot_merged_features(df, fp_signal, contraction_frequency, merge_features_df, case):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(x=df.index, y=df.force, mode="lines", name="Force"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df.calc, mode="lines", name="Calcium"), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=fp_signal.index, y=fp_signal, mode="lines", name="MEA"),
        row=3,
        col=1,
    )

    for i in range(len(merge_features_df)):
        event = merge_features_df.iloc[i]
        is_arrythmia = (
            event["force_arrythmia_prediction"] == 1
            or event["calc_arrythmia_prediction"] == 1
        )
        is_potential_outlier = (
            event["force_potential_outlier"] == 1
            or event["calc_potential_outlier"] == 1
        )
        if is_arrythmia:
            f_peak_color, c_peak_color = "orange", "orange"
        else:
            f_peak_color, c_peak_color = "red", "purple"
        if is_potential_outlier:
            f_marker, c_marker = "x", "x"
        else:
            f_marker, c_marker = "circle", "circle"
        fig.add_trace(
            go.Scatter(
                x=[event["force_peaks_time_idx"]],
                y=[df.force.loc[event["force_peaks_time_idx"]]],
                mode="markers",
                marker=dict(color=f_peak_color, symbol=f_marker, size=5),
                name="Force Peak",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[event["calc_peaks_time_idx"]],
                y=[df.calc.loc[event["calc_peaks_time_idx"]]],
                mode="markers",
                marker=dict(color=c_peak_color, symbol=c_marker, size=5),
                name="Calcium Peak",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[event["start_time"]],
                y=[fp_signal.loc[event["start_time"]]],
                mode="markers",
                marker=dict(size=5, color="black"),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[event["end_time"]],
                y=[fp_signal.loc[event["end_time"]]],
                mode="markers",
                marker=dict(size=5, color="yellow"),
            ),
            row=3,
            col=1,
        )

        # vertical lines for the start and end of the event

        # force
        fig.add_shape(
            dict(
                type="line",
                x0=event["start_time"],
                y0=df.force.min(),
                x1=event["start_time"],
                y1=df.force.max(),
                line=dict(color="rgba(0, 0, 0, 0.5)", width=0.5, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_shape(
            dict(
                type="line",
                x0=event["end_time"],
                y0=df.force.min(),
                x1=event["end_time"],
                y1=df.force.max(),
                line=dict(color="rgba(0, 0, 0, 0.5)", width=0.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # calcium
        fig.add_shape(
            dict(
                type="line",
                x0=event["start_time"],
                y0=df.calc.min(),
                x1=event["start_time"],
                y1=df.calc.max(),
                line=dict(color="rgba(0, 0, 0, 0.5)", width=0.5, dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_shape(
            dict(
                type="line",
                x0=event["end_time"],
                y0=df.calc.min(),
                x1=event["end_time"],
                y1=df.calc.max(),
                line=dict(color="rgba(0, 0, 0, 0.5)", width=0.5, dash="dash"),
            ),
            row=2,
            col=1,
        )

        # mea
        fig.add_shape(
            dict(
                type="line",
                x0=event["start_time"],
                y0=fp_signal.min(),
                x1=event["start_time"],
                y1=fp_signal.max(),
                line=dict(color="rgba(0, 0, 0, 0.5)", width=0.5, dash="dash"),
            ),
            row=3,
            col=1,
        )
        fig.add_shape(
            dict(
                type="line",
                x0=event["end_time"],
                y0=fp_signal.min(),
                x1=event["end_time"],
                y1=fp_signal.max(),
                line=dict(color="rgba(0, 0, 0, 0.5)", width=0.5, dash="dash"),
            ),
            row=3,
            col=1,
        )
    fig.update_layout(
        title=f"{case} - Merged Features [Contraction Frequency: {np.round(contraction_frequency, 2)} Hz]",
        autosize=False,
        width=1000,
        height=1000,
    )
    return fig


def merge_ecc_features(
    data_dir,
    force_peaks_dir,
    field_potential_case_filter,
    dropped_case_file,
    output_dir,
    force_arrythmia_classifier_path,
    calcium_arrythmia_classifier_path,
    raw_data_dir,
):
    cases = [
        f.replace(".hdf", "")
        for f in os.listdir(data_dir)
        if f.endswith(".hdf") and field_potential_case_filter in f
    ]

    arrythmia_classifier = init_arrythmia_classifiers(
        force_arrythmia_classifier_path,
        calcium_arrythmia_classifier_path,
        raw_data_dir,
    )
    if os.path.exists(dropped_case_file):
        dropped_cases = json.load(open(dropped_case_file))
        print("Found list of dropped cases: ", dropped_cases)
        cases = [case for case in cases if case not in dropped_cases]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Plots"), exist_ok=True)
    skipped_cases = []
    for case in cases:
        print("Processing case: ", case)
        df = pd.read_hdf(os.path.join(data_dir, f"{case}.hdf"))
        has_fp = "mea" in df.columns
        skipped_mea = False

        force_feature_extractor = CommonFeatureExtractor(
            case, FORCE_SIGNAL_TYPE, data_dir, force_peaks_dir
        )
        force_features, _ = force_feature_extractor.extract()

        calcium_feature_extractor = CommonFeatureExtractor(
            case, CALCIUM_SIGNAL_TYPE, data_dir, force_peaks_dir
        )
        calcium_features, _ = calcium_feature_extractor.extract()

        if has_fp:
            fp_feature_extractor = FieldPotentialFeatureExtractor(case, data_dir)
            fp_features_result = fp_feature_extractor.extract()
            if fp_features_result is None:
                skipped_mea = True
            else:
                fp_features, fp_signal, contraction_frequency, _ = fp_features_result
        else:
            skipped_mea = True

        if skipped_mea:
            print(
                f"Skipping case {case} due to unavailability of Field Potential signal"
            )
            skipped_cases.append(case)
            continue

        common_records = []
        for i in range(len(fp_features)):
            event = fp_features.iloc[i]
            event_start_time = event["start_time"]
            event_end_time = event["end_time"]

            record = event.to_dict()
            force_peaks_in_event_interval = force_features[
                (force_features["peaks_time_idx"] > event_start_time)
                & (force_features["peaks_time_idx"] <= event_end_time)
            ]

            if force_peaks_in_event_interval.shape[0] == 0:
                continue

            highest_force_peak_in_event_interval = force_peaks_in_event_interval[
                force_peaks_in_event_interval["peak_amplitude"]
                == force_peaks_in_event_interval["peak_amplitude"].max()
            ]

            highest_force_peak_in_event_interval = (
                highest_force_peak_in_event_interval.add_prefix("force_")
            )

            calc_peaks_in_event_interval = calcium_features[
                (calcium_features["peaks_time_idx"] > event_start_time)
                & (calcium_features["peaks_time_idx"] <= event_end_time)
            ]

            if calc_peaks_in_event_interval.shape[0] == 0:
                continue

            record.update(
                highest_force_peak_in_event_interval.to_dict(orient="records")[0]
            )

            highest_calcium_peak_in_event_interval = calc_peaks_in_event_interval[
                calc_peaks_in_event_interval["peak_amplitude"]
                == calc_peaks_in_event_interval["peak_amplitude"].max()
            ]

            highest_calcium_peak_in_event_interval = (
                highest_calcium_peak_in_event_interval.add_prefix("calc_")
            )

            record.update(
                highest_calcium_peak_in_event_interval.to_dict(orient="records")[0]
            )

            force_arrythmia_classification = arrythmia_classifier(
                case,
                df,
                FORCE_SIGNAL_TYPE,
                highest_force_peak_in_event_interval["force_peaks_time_idx"].iloc[0],
            )
            calcium_arrythmia_classification = arrythmia_classifier(
                case,
                df,
                CALCIUM_SIGNAL_TYPE,
                highest_force_peak_in_event_interval["force_peaks_time_idx"].iloc[0],
            )
            record["force_arrythmia_prediction"] = force_arrythmia_classification
            record["calc_arrythmia_prediction"] = calcium_arrythmia_classification
            common_records.append(record)

        common_records_df = pd.DataFrame(
            common_records, index=range(len(common_records))
        )
        # convert all columns with index in their name to integer
        for col in common_records_df.columns:
            if "index" in col:
                common_records_df[col] = pd.to_numeric(
                    common_records_df[col], downcast="integer"
                )

        common_records_df.to_csv(
            os.path.join(output_dir, "Data", f"{case}.csv"), index=False
        )
        fig = plot_merged_features(
            df, fp_signal, contraction_frequency, common_records_df, case
        )
        fig.write_html(os.path.join(output_dir, "Plots", f"{case}.html"))
    if len(skipped_cases) > 0:
        with open(os.path.join(output_dir, "skipped_cases.json"), "w") as f:
            print(
                "Skipped cases due to unavailability of Field Potential signal: ",
                skipped_cases,
            )
            json.dump(skipped_cases, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/syedshameersarwar/Desktop/Work/MyOfarm/Internship/Feature Extracion & Clustering/arrythmia-prod/Preprocessed/HDFs",
        help="Path to the directory containing the preprocessed HDFs",
    )
    parser.add_argument(
        "--force_peaks_dir",
        type=str,
        default="/home/syedshameersarwar/Desktop/Work/MyOfarm/Internship/Feature Extracion & Clustering/arrythmia-prod/Preprocessed/Peaks",
        help="Path to the directory containing the force peaks json file",
    )
    parser.add_argument(
        "--field_potential_case_filter",
        type=str,
        default="run1b",
        help="Filter for the field potential case, Only cases with this filter in their name will be processed",
    )
    parser.add_argument(
        "--dropped_case_file",
        type=str,
        default="./dropped_cases.json",
        help="Path to the  json file containing the list of dropped cases",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Features",
        help="Path to the directory to save the merged features, Features will be saved at <output_dir>/Features/<case_name>.csv, and Plots will be saved at <output_dir>/Plots/<case_name>.html",
    )
    parser.add_argument(
        "--force_arrythmia_classifier_path",
        type=str,
        default="/home/syedshameersarwar/Desktop/Work/MyOfarm/Internship/Feature Extracion & Clustering/arrythmia-prod/Models/force/training_run_2025-03-03_00-13-29/model_best_window_statistical_1.2s.joblib",
        help="Path to the force classifier model",
    )
    parser.add_argument(
        "--calcium_arrythmia_classifier_path",
        type=str,
        default="/home/syedshameersarwar/Desktop/Work/MyOfarm/Internship/Feature Extracion & Clustering/arrythmia-prod/Models/calcium/training_run_2025-03-03_00-13-29/model_best_window_statistical_1.2s.joblib",
        help="Path to the calcium classifier model",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="/home/syedshameersarwar/Desktop/Work/MyOfarm/Internship/Feature Extracion & Clustering/arrythmia-prod/RawHDFs",
        help="Path to the directory containing the raw data",
    )

    args = parser.parse_args()

    merge_ecc_features(
        data_dir=args.data_dir,
        force_peaks_dir=args.force_peaks_dir,
        field_potential_case_filter=args.field_potential_case_filter,
        dropped_case_file=args.dropped_case_file,
        output_dir=args.output_dir,
        force_arrythmia_classifier_path=args.force_arrythmia_classifier_path,
        calcium_arrythmia_classifier_path=args.calcium_arrythmia_classifier_path,
        raw_data_dir=args.raw_data_dir,
    )
